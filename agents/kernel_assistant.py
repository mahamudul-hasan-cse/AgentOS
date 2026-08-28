"""An assistant that runs INSIDE the kernel, as a process, like any other agent.

This is the point of the feature. A chat assistant bolted onto the side of the
project would be a web service that happens to know about AgentOS-Lite. This one
is a process in the scheduler with pid "assistant", USER privilege, and its own
quota, and it learns about the system the only way any agent can: by issuing
syscalls through the dispatcher. Every read it performs is trapped and logged,
so a user watching the syscall trace can see exactly what it looked at before
answering — and if it asks for something its privilege forbids, the kernel
refuses it in public.

Two grounding sources, both reached through syscalls:

- **Live state** via the read-only introspection family (PROC_LIST, MEM_STATE,
  RESOURCE_STATE, SYSCALL_LOG). Nothing about the running system is described
  from memory or inference; if it was not read this turn, the assistant says it
  needs to check.
- **Project specifics** via FILE_SEARCH / FILE_READ over this repository's own
  documentation, indexed into the semantic file system at startup. "How does
  paging work here" is a question about *this* codebase, and a model's general
  knowledge of operating systems is exactly the wrong source for it.

DESIGN DECISION: which syscalls to issue is chosen deterministically from the
question, not by asking the model to pick tools. Two reasons. It keeps the
retrieval step working with no LLM provider configured at all — the syscalls
still fire, still appear in the trace, and still get access-controlled, which is
what the tests and the demo actually exercise. And it means the reads happen
*before* any text is generated, so the model is never in a position to describe
state it has not been handed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kernel.syscalls import Syscall, SyscallDispatcher, SyscallStatus, SyscallType

from .example_agents import KernelAgent

#: the assistant's pid — it is a real entry in the process table under this name
ASSISTANT_PID = "assistant"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: repository documentation indexed into the semantic FS at startup. These are
#: the ground truth for questions about the project itself.
DOC_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("README.md", "readme"),
    ("benchmarks/README.md", "benchmarks"),
    ("PROJECT_PLAN.md", "plan"),
    ("shell/README.md", "shell"),
)

#: markdown is split on headings so a retrieved chunk is a coherent section
#: rather than an arbitrary window; oversized sections are further split.
MAX_CHUNK_CHARS = 1800

#: Keyword hints deciding which of the *optional* introspection reads a question
#: needs. There is no "processes" entry: PROC_LIST is issued unconditionally as
#: baseline context (see _plan), because it is world-readable and effectively
#: free, and keyword-gating it made vaguely-worded questions retrieve nothing.
_LIVE_STATE_HINTS = {
    "memory": (
        "memory", "page", "pages", "paging", "ram", "swap", "resident",
        "evict", "eviction", "context window",
    ),
    "resources": (
        "resource", "provider", "quota", "rate limit", "deadlock", "banker",
        "capacity", "pool", "allocation",
    ),
    "syscalls": (
        "syscall", "syscalls", "trace", "log", "call history", "what have you done",
    ),
}


def _chunk_markdown(text: str, label: str) -> List[Tuple[str, str]]:
    """Split markdown into heading-delimited chunks -> [(filename, content)]."""
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "intro"
    current: List[str] = []
    for line in lines:
        if re.match(r"^#{1,3} ", line):
            if current:
                sections.append((current_title, current))
            current_title = line.lstrip("#").strip()
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append((current_title, current))

    chunks: List[Tuple[str, str]] = []
    for title, body in sections:
        content = "\n".join(body).strip()
        if not content:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "section"
        # split anything still oversized, keeping the heading on each part
        parts = [content[i : i + MAX_CHUNK_CHARS] for i in range(0, len(content), MAX_CHUNK_CHARS)]
        for n, part in enumerate(parts):
            suffix = f"-{n}" if len(parts) > 1 else ""
            chunks.append((f"{label}__{slug}{suffix}.md", f"[source: {label}] {part}"))
    return chunks


class KernelAssistant(KernelAgent):
    """A chat assistant that is also process `assistant` in the scheduler."""

    def __init__(self, dispatcher: SyscallDispatcher):
        super().__init__(name="KernelAssistant", agent_id=ASSISTANT_PID, dispatcher=dispatcher)

    # --- lifecycle --------------------------------------------------------

    async def register(self, priority: int = 1) -> Dict[str, Any]:
        """Register as a real process via SPAWN_AGENT (parent init).

        Idempotent: if ``assistant`` already exists in the process table, ACL
        is refreshed but no second spawn is attempted.
        """
        from kernel.access_control import AgentPrivilege

        scheduler = self.dispatcher.scheduler
        scheduler.ensure_init()
        # init is the hierarchy root; spawning through it matches the intended
        # parent_pid and routes registration through the syscall trap.
        self.dispatcher.acl.registry.register("init", AgentPrivilege.KERNEL)

        existing = scheduler.get(ASSISTANT_PID)
        if existing is None:
            syscall = await self.dispatcher.dispatch(
                "init",
                SyscallType.SPAWN_AGENT,
                pid=ASSISTANT_PID,
                privilege=AgentPrivilege.USER.value,
                estimated_burst=0.0,
                priority=priority,
            )
            if syscall.status != SyscallStatus.SUCCESS:
                raise RuntimeError(
                    f"assistant SPAWN_AGENT failed: {syscall.status.value} {syscall.result}"
                )
        else:
            self.dispatcher.acl.registry.register(ASSISTANT_PID, AgentPrivilege.USER)

        self.dispatcher.quota_manager.usage(ASSISTANT_PID)
        process = scheduler.get(ASSISTANT_PID)
        return {
            "pid": ASSISTANT_PID,
            "state": process.state if process else "unknown",
            "parent_pid": process.parent_pid if process else None,
            "privilege": AgentPrivilege.USER.value,
        }

    def is_alive(self) -> bool:
        """False once the process has been killed from the shell or dashboard."""
        process = self.dispatcher.scheduler.get(ASSISTANT_PID)
        return process is not None and process.state not in ("terminated", "zombie")

    # --- documentation indexing -------------------------------------------

    async def index_documentation(self, repo_root: Optional[Path] = None) -> Dict[str, Any]:
        """Index the repository's own docs into the semantic FS via FILE_WRITE.

        Done through syscalls rather than by touching SemanticFS directly, so
        even the assistant's setup is visible in the trace and subject to the
        same access control as everything else.
        """
        root = Path(repo_root) if repo_root is not None else _REPO_ROOT
        indexed: List[str] = []
        missing: List[str] = []
        for relative, label in DOC_SOURCES:
            path = root / relative
            if not path.is_file():
                missing.append(relative)
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001 — a doc we cannot read is not fatal
                missing.append(relative)
                continue
            for filename, content in _chunk_markdown(text, label):
                syscall = await self.dispatcher.dispatch(
                    self.agent_id,
                    SyscallType.FILE_WRITE,
                    filename=filename,
                    content=content,
                )
                if syscall.status == SyscallStatus.SUCCESS:
                    indexed.append(filename)
        return {"indexed": len(indexed), "files": indexed, "missing": missing}

    # --- retrieval --------------------------------------------------------

    def _known_agent_ids(self) -> List[str]:
        return [p.pid for p in self.dispatcher.scheduler.queue]

    def _foreign_memory_target(self, message: str) -> Optional[str]:
        """Detect a request for ANOTHER agent's memory, e.g. "what is in
        researcher's memory". Matched against real pids so ordinary prose does
        not accidentally trigger a permission probe."""
        lowered = message.lower()
        if not any(word in lowered for word in ("memory", "pages", "paging", "ram", "swap")):
            return None
        for pid in self._known_agent_ids():
            if pid in (ASSISTANT_PID, "init"):
                continue
            if re.search(rf"\b{re.escape(pid.lower())}\b", lowered):
                return pid
        return None

    def _plan(self, message: str) -> List[Tuple[SyscallType, Dict[str, Any], str]]:
        """Decide which reads this question needs. Deterministic by design."""
        lowered = message.lower()
        plan: List[Tuple[SyscallType, Dict[str, Any], str]] = []

        def wants(category: str) -> bool:
            return any(h in lowered for h in _LIVE_STATE_HINTS[category])

        # PROC_LIST runs unconditionally. It is world-readable and costs ~0.1ms,
        # and it is the baseline context for almost any question about the
        # running system. Gating it on keywords meant a loosely-worded question
        # ("how busy is it?") retrieved nothing and the assistant then correctly
        # but uselessly reported that it had no state to answer from — refusing
        # over imprecise phrasing rather than over genuinely missing data.
        plan.append((SyscallType.PROC_LIST, {}, "process table"))

        foreign = self._foreign_memory_target(message)
        if foreign is not None:
            # Deliberately attempted. USER privilege forbids it, and the refusal
            # is the demonstration -- the assistant then explains the ACL.
            plan.append(
                (SyscallType.MEM_STATE, {"target_agent_id": foreign}, f"{foreign} memory")
            )
        elif wants("memory"):
            plan.append((SyscallType.MEM_STATE, {}, "own memory"))

        if wants("resources"):
            plan.append((SyscallType.RESOURCE_STATE, {}, "resources + deadlock"))
        if wants("syscalls"):
            plan.append((SyscallType.SYSCALL_LOG, {"limit": 15}, "own syscall trace"))

        # Documentation retrieval always runs: almost every question has a
        # "how does this project do it" component, and grounding that in the
        # repo's docs is the whole point of indexing them.
        plan.append((SyscallType.FILE_SEARCH, {"query": message, "top_k": 3}, "docs"))
        return plan

    async def gather(self, message: str) -> Tuple[List[Dict[str, Any]], List[Syscall]]:
        """Issue the planned syscalls. Returns (observations, syscall records)."""
        observations: List[Dict[str, Any]] = []
        records: List[Syscall] = []

        for syscall_type, kwargs, label in self._plan(message):
            syscall = await self.dispatcher.dispatch(self.agent_id, syscall_type, **kwargs)
            records.append(syscall)
            observations.append(
                {
                    "syscall": syscall_type.value,
                    "target": label,
                    "status": syscall.status.value,
                    "result": syscall.result,
                }
            )

            # A FILE_SEARCH only returns 160-char snippets, so pull the full text
            # of the best matches. Another syscall, another trace entry.
            if syscall_type == SyscallType.FILE_SEARCH and syscall.status == SyscallStatus.SUCCESS:
                for match in (syscall.result or {}).get("results", [])[:2]:
                    read = await self.dispatcher.dispatch(
                        self.agent_id,
                        SyscallType.FILE_READ,
                        filename=match["filename"],
                    )
                    records.append(read)
                    observations.append(
                        {
                            "syscall": SyscallType.FILE_READ.value,
                            "target": match["filename"],
                            "status": read.status.value,
                            "result": read.result,
                        }
                    )
        return observations, records

    # --- prompting --------------------------------------------------------

    SYSTEM_RULES = """You are the AgentOS-Lite kernel assistant. You are not an
external service: you are process "assistant" running inside this kernel, at
USER privilege, with your own quota, and you obtained everything below by
issuing syscalls through the kernel's syscall dispatcher this turn.

Rules you must follow exactly:

1. GROUND EVERY CLAIM ABOUT CURRENT SYSTEM STATE in the SYSCALL RESULTS below.
   Those results are the only thing you know about the running system. Never
   describe kernel state you did not read. If the answer genuinely needs state
   that is not in the results, say you need to check and name the syscall you
   would issue — but read rule 2 first, because that is a narrower case than it
   sounds.

   These are the ONLY syscalls that exist. Never name one outside this list:
     PROC_LIST       the process table and hierarchy
     MEM_STATE       an agent's paged memory (RAM vs swap) and its quota usage
     RESOURCE_STATE  provider rate-limit pools and deadlock status
     SYSCALL_LOG     an agent's own recent syscalls
     FILE_SEARCH     semantic search over indexed documentation
     FILE_READ       read one indexed document in full
     LLM_CALL        generate text
     MEM_READ / MEM_WRITE / FILE_WRITE / IPC_SEND / IPC_RECV /
     BLACKBOARD_READ / BLACKBOARD_WRITE / SPAWN_AGENT / WAIT / TERMINATE_AGENT
   Inventing a syscall name is a serious error: it would tell the user this
   kernel has a capability it does not have.

2. ANSWER FROM WHAT YOU RETRIEVED — do not hedge when you already have the
   data. If the syscall results plausibly answer the question under a
   reasonable reading, then in this order:
     (a) give the answer, with the concrete values;
     (b) add ONE short clause, in your own words, naming the reading you took
         — for instance a question about "how much memory" might be answered
         "counting resident pages only". Exactly one such clause per answer:
         do not restate it, and do not copy the wording of this example;
     (c) only then, and only if a materially different reading exists, offer
         it in one sentence.

   Never open by asking which interpretation was meant, and never reply with
   only a request for clarification when the results in front of you would
   answer any reasonable version of the question. Imprecise phrasing is not
   missing data: resolve it, say how you resolved it, and move on.

   Refusing to answer is right in exactly one case: the data is genuinely not
   in the results above — no syscall this turn retrieved it, or one was denied.
   That is rule 1's territory, and rule 1 still binds absolutely: a gap is
   never to be filled with a guess or with general knowledge.

3. NEVER explain this project's specifics from general knowledge about
   operating systems. Questions about how paging, Semantic-LRU, scheduling,
   deadlock handling or the benchmarks work HERE must be answered from the
   RETRIEVED DOCUMENTATION below, which is this repository's own docs. If the
   documentation does not cover it, say so rather than filling the gap from
   general knowledge. Textbook OS facts may be used only to explain a concept
   you have already grounded in a retrieved document, and never to assert what
   this codebase does.

4. IF A SYSCALL WAS DENIED, that is a feature working, not a failure. Explain
   what was refused and WHY in terms of the kernel's access control: you run at
   USER privilege, and USER agents may read only their own memory and their own
   syscall trace; crossing into another agent requires KERNEL privilege. Do not
   apologise for it, do not retry it, and above all do not invent the data you
   were refused.

5. If a syscall returned QUOTA_EXCEEDED, explain that you are subject to
   per-agent quotas exactly like any other agent, and say which quota was hit.
   Only say a quota was exceeded if a syscall actually came back
   QUOTA_EXCEEDED. The `quota` block inside a MEM_STATE result is usage
   reporting, not a failure — do not read it as one.

6. Be concise and concrete. Prefer citing actual values you read (pids, states,
   page counts, latencies) over general description."""

    def _build_prompt(
        self,
        message: str,
        observations: Sequence[Dict[str, Any]],
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> str:
        live: List[str] = []
        docs: List[str] = []
        denied: List[str] = []

        for obs in observations:
            kind, status, target = obs["syscall"], obs["status"], obs["target"]
            if status != SyscallStatus.SUCCESS.value:
                error = (obs["result"] or {}).get("error", status)
                denied.append(f"- {kind} ({target}) -> {status.upper()}: {error}")
                continue
            if kind in (SyscallType.FILE_SEARCH.value, SyscallType.FILE_READ.value):
                if kind == SyscallType.FILE_READ.value:
                    docs.append(
                        f"--- {obs['result'].get('filename')} ---\n"
                        f"{obs['result'].get('content', '')}"
                    )
            else:
                live.append(f"- {kind} ({target}):\n{_summarise(obs['result'])}")

        sections = [self.SYSTEM_RULES, ""]
        if history:
            turns = "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]
            )
            sections += ["CONVERSATION SO FAR:", turns, ""]
        sections += [
            "SYSCALL RESULTS (live kernel state you read this turn):",
            "\n".join(live) if live else "(none — you issued no live-state reads this turn)",
            "",
            "DENIED OR FAILED SYSCALLS (explain these per rules 3 and 4):",
            "\n".join(denied) if denied else "(none)",
            "",
            "RETRIEVED DOCUMENTATION (this repository's own docs):",
            "\n\n".join(docs) if docs else "(no documentation matched this question)",
            "",
            f"USER QUESTION: {message}",
            "",
            "Answer following the rules above.",
        ]
        return "\n".join(sections)

    # --- the public entry point -------------------------------------------

    async def answer(
        self,
        message: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        driver: str = "groq",
    ) -> Dict[str, Any]:
        """Answer one question. Returns the answer plus every syscall issued."""
        if not self.is_alive():
            # The process was killed. An assistant that kept answering after its
            # own process was terminated would be pretending to be part of the
            # system while standing outside it.
            return {
                "answer": (
                    f"Process '{ASSISTANT_PID}' is not running — it has been terminated. "
                    f"I cannot issue syscalls without a live process. Restart it with "
                    f"POST /assistant/restart to continue."
                ),
                "syscalls": [],
                "process_alive": False,
                "process_state": self._process_state(),
            }

        observations, records = await self.gather(message)
        prompt = self._build_prompt(message, observations, history)

        llm = await self.dispatcher.dispatch(
            self.agent_id, SyscallType.LLM_CALL, prompt=prompt, driver=driver
        )
        records.append(llm)

        if llm.status == SyscallStatus.SUCCESS:
            answer = llm.result.get("text", "")
        else:
            # No provider (or quota exhausted). Report it honestly and still hand
            # back the grounding — the syscalls really were issued, and for
            # QUOTA_EXCEEDED the refusal is itself the thing worth showing.
            error = (llm.result or {}).get("error", "unknown error")
            answer = (
                f"I could not generate an answer: the LLM_CALL syscall returned "
                f"{llm.status.value.upper()} ({error}).\n\n"
                f"I did complete {len(records) - 1} read syscall(s) for this question — "
                f"they are listed below, and their results are what any answer would "
                f"have been grounded in."
            )
            if llm.status == SyscallStatus.QUOTA_EXCEEDED:
                answer += (
                    "\n\nThis is the kernel's per-agent quota system working as intended: "
                    "I am process 'assistant' and am rate-limited exactly like any other "
                    "agent, with no special privilege."
                )

        return {
            "answer": answer,
            "syscalls": [_syscall_summary(s) for s in records],
            "process_alive": True,
            "process_state": self._process_state(),
        }

    def _process_state(self) -> Optional[str]:
        process = self.dispatcher.scheduler.get(ASSISTANT_PID)
        return process.state if process is not None else None


def _syscall_summary(syscall: Syscall) -> Dict[str, Any]:
    """What the dashboard shows beneath an answer: type, target, latency."""
    args = syscall.args or {}
    target = (
        args.get("target_agent_id")
        or args.get("filename")
        or args.get("query")
        or syscall.agent_id
    )
    if isinstance(target, str) and len(target) > 60:
        target = target[:57] + "..."
    return {
        "syscall_id": syscall.syscall_id,
        "type": syscall.type.value,
        "target": target,
        "status": syscall.status.value,
        "latency_ms": round(syscall.latency_ms, 2) if syscall.latency_ms is not None else None,
        "error": (syscall.result or {}).get("error")
        if syscall.status != SyscallStatus.SUCCESS and isinstance(syscall.result, dict)
        else None,
    }


def _summarise(result: Any, limit: int = 1200) -> str:
    """Compact a syscall result for the prompt without hiding the numbers."""
    if result is None:
        return "  (no result)"
    if isinstance(result, dict) and "processes" in result:
        # Every field is labelled. An earlier unlabelled form ("init state=ready
        # ...") led the model to report that processes had no PID, because it
        # could not tell which token was the identifier.
        rows = [
            f"  pid={p['pid']} state={p['state']} parent_pid={p['parent_pid']} "
            f"priority={p['priority']} remaining_burst={p['remaining_burst']}"
            for p in result["processes"]
        ]
        header = f"  ({len(result['processes'])} processes; pid is the identifier)"
        return "\n".join([header] + rows) if rows else "  (no processes)"
    text = repr(result)
    return "  " + (text if len(text) <= limit else text[:limit] + " …(truncated)")
