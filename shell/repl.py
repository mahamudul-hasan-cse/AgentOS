#!/usr/bin/env python3
"""AgentOS-Lite interactive shell — a Unix-style REPL over the kernel's HTTP API.

Run against a live FastAPI backend (see api/main.py):

    python shell/repl.py                       # connect to http://localhost:8000 as root
    python shell/repl.py --url http://host:8000 --agent alice

Everything the shell does maps onto an existing endpoint. Command *resolution*
(parsing) is kept free of network calls so it can be unit-tested in isolation;
the handlers below are the only part that touches the backend.

Standard library only, plus `requests` (already a project dependency).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import requests

DEFAULT_URL = "http://localhost:8000"
DEFAULT_AGENT = "root"
REQUEST_TIMEOUT = 15
TOP_INTERVAL = 2.0

# HTTP status -> friendly prefix, so users never see a raw status or traceback.
STATUS_MESSAGES = {
    400: "bad request",
    403: "permission denied",
    404: "not found",
    429: "quota exceeded",
    501: "not implemented",
    502: "backend/provider error",
    503: "service unavailable",
}


class ShellError(Exception):
    """A user-facing error. The REPL prints these as a clean one-liner."""


@dataclass
class Context:
    """Everything a handler needs: where the backend is, who we are, and a
    reusable HTTP session."""

    base_url: str
    agent: str
    session: requests.Session = field(default_factory=requests.Session)


# --------------------------------------------------------------------------
# HTTP helper — maps transport/status failures onto clean ShellError messages
# --------------------------------------------------------------------------

def _detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except ValueError:
        pass
    return (resp.text or "").strip()[:200]


def api(ctx: Context, method: str, path: str, **kwargs) -> object:
    """Call the backend and return parsed JSON, or raise a friendly ShellError."""
    url = ctx.base_url.rstrip("/") + path
    try:
        resp = ctx.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        raise ShellError(f"cannot connect to backend at {ctx.base_url} (is it running?)")
    except requests.exceptions.Timeout:
        raise ShellError("request timed out")
    except requests.exceptions.RequestException as exc:
        raise ShellError(f"request failed: {exc}")

    if not resp.ok:
        prefix = STATUS_MESSAGES.get(resp.status_code, f"error {resp.status_code}")
        detail = _detail(resp)
        raise ShellError(f"{prefix}: {detail}" if detail else prefix)

    if resp.status_code == 204 or not resp.content:
        return None
    try:
        return resp.json()
    except ValueError:
        raise ShellError("backend returned a non-JSON response")


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------

def format_table(headers: List[str], rows: List[List[object]]) -> str:
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# --------------------------------------------------------------------------
# Command handlers — each takes (ctx, args) and prints its own output
# --------------------------------------------------------------------------

PS_HEADERS = ["PID", "PPID", "STATE", "ARRIVAL", "REMAINING", "PRIO", "EXIT"]


def _process_rows(state: dict) -> List[List[object]]:
    rows = []
    for p in state.get("processes", []):
        exit_status = p.get("exit_status")
        rows.append([
            p["pid"],
            p.get("parent_pid") or "-",
            p["state"] + (" <defunct>" if p["state"] == "zombie" else ""),
            p["arrival_time"],
            p["remaining_burst"],
            p["priority"],
            "-" if exit_status is None else exit_status,
        ])
    return rows


def handle_ps(ctx: Context, args: List[str]) -> None:
    state = api(ctx, "GET", "/scheduler/state")
    rows = _process_rows(state)
    if not rows:
        print("no processes in the queue")
        return
    print(format_table(PS_HEADERS, rows))
    zombies = sum(1 for p in state.get("processes", []) if p["state"] == "zombie")
    if zombies:
        print(
            f"\n{zombies} zombie process(es) awaiting reap "
            f"(a parent clears one with: wait <parent> <child>)"
        )


def handle_pstree(ctx: Context, args: List[str]) -> None:
    """Render the process hierarchy with ASCII indentation, like pstree(1)."""
    tree = api(ctx, "GET", "/scheduler/tree")
    if not tree:
        print("no process tree (is the backend seeded?)")
        return

    def label(node: dict) -> str:
        state = node.get("state", "?")
        if state == "zombie":
            exit_status = node.get("exit_status")
            suffix = "" if exit_status is None else f" exit={exit_status}"
            return f"{node['pid']}  [zombie <defunct>{suffix}]"
        return f"{node['pid']}  ({state})"

    def walk(node: dict, prefix: str = "", is_last: bool = True, root: bool = True) -> None:
        if root:
            print(label(node))
        else:
            print(f"{prefix}{'`-- ' if is_last else '|-- '}{label(node)}")
            prefix += "    " if is_last else "|   "
        children = node.get("children", [])
        for i, child in enumerate(children):
            walk(child, prefix, i == len(children) - 1, root=False)

    walk(tree)


def handle_wait(ctx: Context, args: List[str]) -> None:
    """wait <parent> [child] — reap a zombie child and read its exit status."""
    parent = args[0]
    child = args[1] if len(args) > 1 else None
    params = {"child_pid": child} if child else {}
    result = api(ctx, "POST", f"/scheduler/wait/{parent}", params=params)
    if not result.get("reaped"):
        target = f" '{child}'" if child else ""
        print(f"no zombie child{target} to reap for '{parent}'")
        return
    print(f"reaped '{result['pid']}' (exit status {result['exit_status']})")


def handle_spawn(ctx: Context, args: List[str]) -> None:
    """spawn [pid] — fork a child process owned by the shell's agent."""
    payload = {"agent_id": ctx.agent}
    if args:
        payload["pid"] = args[0]
    result = api(ctx, "POST", "/scheduler/spawn", json=payload)
    print(
        f"spawned '{result['pid']}' (parent={result['parent_pid']}, "
        f"privilege={result['privilege']})"
    )


def handle_deadlock(ctx: Context, args: List[str]) -> None:
    """deadlock [detect] — show the wait-for graph and cycle status."""
    force = bool(args) and args[0] in ("detect", "-d", "--detect")
    if force:
        api(ctx, "POST", "/deadlock/detect")

    status = api(ctx, "GET", "/deadlock/status")
    graph = api(ctx, "GET", "/deadlock/graph")

    mode = "avoidance (Banker's)" if status["avoidance_enabled"] else "detection + recovery"
    print(f"strategy: {mode}   detection runs: {status['detection_runs']}   "
          f"recoveries: {status['recoveries']}")

    if status["deadlocked"]:
        print(f"\n  ** DEADLOCK ** cycle: {' -> '.join(status['cycle'])} -> {status['cycle'][0]}")
    else:
        print("\n  no deadlock detected")
        if status["avoidance_enabled"]:
            print("  (avoidance is on, so cycles should never form; "
                  "use 'mode off' to allow them for a demo)")

    nodes = graph.get("nodes", [])
    if not nodes:
        print("\n  wait-for graph is empty (no resources held or awaited)")
        return

    rows = []
    for node in nodes:
        holds = ", ".join(f"{p}:{u}" for p, u in (node["holds"] or {}).items()) or "-"
        waits = ", ".join(node["waiting_on"]) or "-"
        marker = " <-- in cycle" if node["agent_id"] in status["cycle"] else ""
        rows.append([node["agent_id"] + marker, holds, waits])
    print()
    print(format_table(["AGENT", "HOLDS", "WAITING ON"], rows))


def handle_mode(ctx: Context, args: List[str]) -> None:
    """mode <on|off> — toggle deadlock avoidance (Banker's Algorithm)."""
    choice = args[0].lower()
    if choice not in ("on", "off"):
        raise ShellError("usage: mode <on|off>")
    result = api(
        ctx, "POST", "/resources/mode", json={"avoidance_enabled": choice == "on"}
    )
    print(f"avoidance {'ENABLED' if result['avoidance_enabled'] else 'DISABLED'} "
          f"-> strategy: {result['strategy']}")
    if not result["avoidance_enabled"]:
        print("  greedy granting: real deadlocks can now form (that's the point)")


def _resource_header(ctx: Context) -> str:
    providers = api(ctx, "GET", "/resources/state").get("providers", {})
    parts = [
        f"{name} {p['allocated']}/{p['total']}" + ("" if p.get("safe", True) else "!")
        for name, p in providers.items()
    ]
    return "providers:  " + "   ".join(parts) if parts else "providers: (none)"


def handle_top(ctx: Context, args: List[str]) -> None:
    print("(top — refreshing every 2s, press Ctrl+C to stop)")
    try:
        while True:
            state = api(ctx, "GET", "/scheduler/state")
            header = _resource_header(ctx)
            rows = _process_rows(state)
            _clear_screen()
            print(f"AgentOS-Lite  {time.strftime('%H:%M:%S')}   {header}\n")
            if rows:
                print(format_table(PS_HEADERS, rows))
            else:
                print("no processes in the queue")
            print("\n(press Ctrl+C to stop)")
            time.sleep(TOP_INTERVAL)
    except KeyboardInterrupt:
        print("\nstopped.")


def handle_kill(ctx: Context, args: List[str]) -> None:
    """kill [-t] <pid> — terminate a process; -t also kills all its descendants."""
    tree = False
    if args and args[0] in ("-t", "--tree"):
        tree = True
        args = args[1:]
    # the parser allows up to two tokens so "-t <pid>" fits; anything else that
    # leaves more than one positional is a mistake and is rejected here.
    if len(args) != 1:
        raise ShellError("usage: kill [-t] <pid>")
    pid = args[0]

    path = f"/scheduler/kill-tree/{pid}" if tree else f"/scheduler/terminate/{pid}"
    result = api(ctx, "POST", path, params={"agent_id": ctx.agent})

    if not result.get("process_found") and not result.get("cancelled_llm_call"):
        print(f"no such process '{pid}' (nothing to kill)")
        return

    if tree:
        killed = result.get("killed", [])
        print(f"killed subtree rooted at '{pid}': {', '.join(killed)}")
        return

    extra = " (cancelled in-flight LLM call)" if result.get("cancelled_llm_call") else ""
    print(f"killed '{pid}'{extra}")
    if result.get("zombie"):
        print(
            f"  '{pid}' is now a zombie holding exit status "
            f"{result.get('exit_status')} until its parent reaps it"
        )
    reparented = result.get("reparented_to_init") or []
    if reparented:
        print(f"  reparented to init (orphans survive): {', '.join(reparented)}")


def handle_limits(ctx: Context, args: List[str]) -> None:
    agent = args[0] if args else ctx.agent
    q = api(ctx, "GET", f"/quotas/{agent}")
    print(f"quotas for '{agent}':")
    print(
        format_table(
            ["RESOURCE", "USED", "LIMIT"],
            [
                ["memory pages", q["pages_used"], q["max_pages"]],
                [
                    f"LLM calls / {int(q['window_seconds'])}s",
                    q["calls_in_window"],
                    q["max_calls_per_minute"],
                ],
            ],
        )
    )


def handle_ls(ctx: Context, args: List[str]) -> None:
    agent = args[0] if args else ctx.agent
    files = api(ctx, "GET", f"/fs/list/{agent}").get("files", [])
    if not files:
        print(f"no files for '{agent}'")
        return
    for name in files:
        print(name)


def handle_cat(ctx: Context, args: List[str]) -> None:
    filename = args[0]
    result = api(
        ctx, "GET", "/fs/read", params={"agent_id": ctx.agent, "filename": filename}
    )
    print(result["content"])


def handle_find(ctx: Context, args: List[str]) -> None:
    query = args[0]
    result = api(ctx, "POST", "/fs/search", json={"agent_id": ctx.agent, "query": query})
    results = result.get("results", [])
    if not results:
        print("no matches")
        return
    rows = [[r["filename"], f"{r['score']:.3f}", r["snippet"].replace("\n", " ")[:60]] for r in results]
    print(format_table(["FILE", "SCORE", "SNIPPET"], rows))


def handle_mem(ctx: Context, args: List[str]) -> None:
    agent = args[0]
    state = api(ctx, "GET", f"/memory/state/{agent}")
    print(
        f"memory for '{agent}':  {state['ram_tokens_used']}/{state['ram_budget_tokens']} "
        f"tokens in RAM"
    )
    ram = state.get("ram_pages", [])
    swapped = state.get("swapped_pages", [])

    def page_rows(pages):
        return [[p["page_id"], p.get("token_count", "?")] for p in pages]

    print(f"\nIN RAM ({len(ram)}):")
    print(format_table(["PAGE", "TOKENS"], page_rows(ram)) if ram else "  (empty)")
    print(f"\nSWAPPED -> ChromaDB ({len(swapped)}):")
    print(format_table(["PAGE", "TOKENS"], page_rows(swapped)) if swapped else "  (empty)")


def handle_strace(ctx: Context, args: List[str]) -> None:
    n = 20
    if args:
        try:
            n = int(args[0])
        except ValueError:
            raise ShellError(f"strace: '{args[0]}' is not a number")
    syscalls = api(ctx, "GET", "/syscalls/log", params={"limit": n}).get("syscalls", [])
    if not syscalls:
        print("no syscalls logged yet")
        return
    rows = []
    for s in syscalls:
        ts = time.strftime("%H:%M:%S", time.localtime(s["timestamp"]))
        lat = f"{s['latency_ms']:.1f}ms" if s.get("latency_ms") is not None else "-"
        rows.append([ts, s["agent_id"], s["type"], s["status"], lat])
    print(format_table(["TIME", "AGENT", "SYSCALL", "STATUS", "LATENCY"], rows))


def handle_run(ctx: Context, args: List[str]) -> None:
    prompt = args[0]
    result = api(
        ctx, "POST", "/generate", json={"prompt": prompt, "agent_id": ctx.agent}
    )
    print(f"[{result['driver_used']}] {result['text']}")


def handle_help(ctx: Context, args: List[str]) -> None:
    print("commands:")
    for name in COMMAND_ORDER:
        cmd = COMMANDS[name]
        print(f"  {cmd.usage:<22} {cmd.help}")


def handle_exit(ctx: Context, args: List[str]) -> None:
    raise _ExitRepl()


class _ExitRepl(Exception):
    """Signals the REPL loop to stop."""


# --------------------------------------------------------------------------
# Command registry + resolution (pure — no network, unit-tested)
# --------------------------------------------------------------------------

# arg_style:
#   "none"   -> takes no arguments
#   "tokens" -> whitespace-split args (min/max enforced)
#   "rest"   -> the remainder of the line is a single argument (for prompts/queries)
@dataclass
class Command:
    name: str
    handler: Callable[[Context, List[str]], None]
    arg_style: str
    min_args: int
    max_args: Optional[int]
    usage: str
    help: str


COMMANDS: dict = {}
COMMAND_ORDER: List[str] = []


def _register(cmd: Command) -> None:
    COMMANDS[cmd.name] = cmd
    COMMAND_ORDER.append(cmd.name)


_register(Command("ps", handle_ps, "none", 0, 0, "ps", "process table"))
_register(Command("top", handle_top, "none", 0, 0, "top", "auto-refreshing ps + provider state"))
_register(Command("kill", handle_kill, "tokens", 1, 2, "kill [-t] <pid>", "terminate a process (-t = whole subtree)"))
_register(Command("pstree", handle_pstree, "none", 0, 0, "pstree", "process hierarchy as a tree"))
_register(Command("spawn", handle_spawn, "tokens", 0, 1, "spawn [pid]", "fork a child process"))
_register(Command("wait", handle_wait, "tokens", 1, 2, "wait <parent> [child]", "reap a zombie child"))
_register(Command("limits", handle_limits, "tokens", 0, 1, "limits [agent]", "quota usage vs limit"))
_register(Command("ls", handle_ls, "tokens", 0, 1, "ls [agent]", "list an agent's files"))
_register(Command("cat", handle_cat, "rest", 1, 1, "cat <filename>", "print a file's contents"))
_register(Command("find", handle_find, "rest", 1, 1, "find <query>", "semantic file search"))
_register(Command("mem", handle_mem, "tokens", 1, 1, "mem <agent>", "RAM vs swapped pages"))
_register(Command("deadlock", handle_deadlock, "tokens", 0, 1, "deadlock [detect]", "wait-for graph + cycle status"))
_register(Command("mode", handle_mode, "tokens", 1, 1, "mode <on|off>", "toggle deadlock avoidance"))
_register(Command("strace", handle_strace, "tokens", 0, 1, "strace [n]", "recent syscalls (default 20)"))
_register(Command("run", handle_run, "rest", 1, 1, "run <prompt>", "LLM_CALL via /generate"))
_register(Command("help", handle_help, "none", 0, 0, "help", "show this help"))
_register(Command("exit", handle_exit, "none", 0, 0, "exit", "leave the shell"))
_register(Command("quit", handle_exit, "none", 0, 0, "quit", "leave the shell"))


@dataclass
class Resolution:
    command: Optional[Command] = None
    args: List[str] = field(default_factory=list)
    error: Optional[str] = None
    empty: bool = False

    @property
    def ok(self) -> bool:
        return self.command is not None and self.error is None


def resolve(line: str) -> Resolution:
    """Parse a command line into a Resolution. Never raises: malformed input
    comes back as a Resolution with `.error` set and `.ok == False`."""
    stripped = line.strip()
    if not stripped:
        return Resolution(empty=True)

    head, _, remainder = stripped.partition(" ")
    remainder = remainder.strip()
    cmd = COMMANDS.get(head)
    if cmd is None:
        return Resolution(error=f"unknown command '{head}' (type 'help' for the list)")

    if cmd.arg_style == "none":
        args: List[str] = []
        if remainder:
            return Resolution(command=cmd, error=f"'{cmd.name}' takes no arguments; usage: {cmd.usage}")
    elif cmd.arg_style == "rest":
        args = [remainder] if remainder else []
    else:  # tokens
        args = remainder.split() if remainder else []

    if len(args) < cmd.min_args or (cmd.max_args is not None and len(args) > cmd.max_args):
        return Resolution(command=cmd, error=f"usage: {cmd.usage}")

    return Resolution(command=cmd, args=args)


# --------------------------------------------------------------------------
# REPL loop
# --------------------------------------------------------------------------

def run_repl(ctx: Context) -> None:
    print(f"AgentOS-Lite shell  -  {ctx.base_url}  (agent: {ctx.agent})")
    print("type 'help' for commands, 'exit' to quit.\n")
    while True:
        try:
            line = input(f"aios:{ctx.agent}$ ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        res = resolve(line)
        if res.empty:
            continue
        if res.error is not None:
            print(res.error)
            continue

        try:
            res.command.handler(ctx, res.args)
        except _ExitRepl:
            break
        except ShellError as exc:
            print(exc)
        except KeyboardInterrupt:
            print()
        except Exception as exc:  # never leak a traceback into the shell
            print(f"unexpected error: {exc}")

    print("bye.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AgentOS-Lite interactive shell")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"backend base URL (default {DEFAULT_URL})")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help=f"identity the shell acts as (default '{DEFAULT_AGENT}', KERNEL-privileged)",
    )
    ns = parser.parse_args(argv)
    ctx = Context(base_url=ns.url, agent=ns.agent)
    run_repl(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
