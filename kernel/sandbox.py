"""Course-project safeguards for executing generated Python code.

This is deliberately modest and honestly scoped. It is NOT a hardened security
sandbox: Python code still runs as a local subprocess under the current user.
The safeguards here reduce obvious blast radius for the demo by using a scratch
working directory, no shell, isolated Python mode, a minimal environment,
captured output, an enforced timeout, and a small AST deny-list for dangerous
imports/calls. Real untrusted-code execution should use an OS/container
sandbox, seccomp/AppArmor, job objects, or a separate locked-down service.
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_TIMEOUT_SECONDS = 3.0
MAX_OUTPUT_CHARS = 4000
KILL_TIMEOUT_SECONDS = 2.0

BLOCKED_IMPORT_ROOTS = {
    "asyncio",
    "ftplib",
    "glob",
    "http",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
BLOCKED_CALLS = {"compile", "eval", "exec", "input", "open", "__import__"}


class SandboxRejected(ValueError):
    """Raised when code fails the conservative pre-execution checks."""


def _validate_python(code: str) -> List[str]:
    try:
        tree = ast.parse(code, filename="<generated>")
    except SyntaxError as exc:
        raise SandboxRejected(f"syntax error: {exc}") from exc

    violations: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in BLOCKED_IMPORT_ROOTS:
                    violations.append(f"blocked import: {name}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                violations.append(f"blocked call: {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in BLOCKED_CALLS:
                violations.append(f"blocked call: {func.attr}()")

    return violations


def _sandbox_subprocess_env() -> Dict[str, str]:
    """Environment for the isolated child interpreter.

    ``-I`` ignores inherited variables, so we pass an explicit minimal set.
    Python 3.10 (notably on Windows, and some Linux CI images) can fail to
    start with only UTF-8 flags unless hash-seed and OS bootstrap vars exist.
    """
    env: Dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONHASHSEED": "0",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    else:
        for key in ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def run_python_sandbox(code: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Execute Python code with demo-grade subprocess safeguards."""
    started = time.perf_counter()
    violations = _validate_python(code)
    if violations:
        return {
            "passed": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "\n".join(violations),
            "timeout": False,
            "rejected": True,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "sandbox": _sandbox_description(timeout_seconds),
        }

    with tempfile.TemporaryDirectory(prefix="agentos_pipeline_") as scratch:
        script = Path(scratch) / "generated_task.py"
        script.write_text(code, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(script)],
            cwd=scratch,
            env=_sandbox_subprocess_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            **_process_group_kwargs(),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            return {
                "passed": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": (stdout or "")[-MAX_OUTPUT_CHARS:],
                "stderr": (stderr or "")[-MAX_OUTPUT_CHARS:],
                "timeout": False,
                "rejected": False,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "sandbox": _sandbox_description(timeout_seconds),
            }
        except subprocess.TimeoutExpired as exc:
            kill_result = _kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=KILL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return {
                "passed": False,
                "exit_code": None,
                "stdout": (stdout or exc.stdout or "")[-MAX_OUTPUT_CHARS:],
                "stderr": (stderr or exc.stderr or "")[-MAX_OUTPUT_CHARS:],
                "timeout": True,
                "timeout_kill": kill_result,
                "rejected": False,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "sandbox": _sandbox_description(timeout_seconds),
            }


def _sandbox_description(timeout_seconds: float) -> Dict[str, Any]:
    return {
        "kind": "restricted_subprocess",
        "timeout_seconds": timeout_seconds,
        "shell": False,
        "cwd": "temporary scratch directory",
        "python_flags": ["-I", "-B"],
        "timeout_kill": "new process group/session is explicitly terminated on expiry",
        "network": "not explicitly granted; blocked only by conservative import checks",
        "limitations": (
            "Not a full security sandbox. It reduces accidental damage for this "
            "course demo but should not run hostile code in production."
        ),
    }


def _process_group_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> Dict[str, Any]:
    """Terminate the spawned process group/tree and report what was attempted."""
    if os.name == "nt":
        try:
            taskkill = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                timeout=KILL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            return {
                "method": "taskkill /F /T timed out; proc.kill fallback",
                "pid": proc.pid,
                "ok": False,
            }
        if taskkill.returncode == 0:
            return {"method": "taskkill /F /T", "pid": proc.pid, "ok": True}
        proc.kill()
        return {
            "method": "taskkill /F /T then proc.kill fallback",
            "pid": proc.pid,
            "ok": False,
            "stderr": (taskkill.stderr or "")[-MAX_OUTPUT_CHARS:],
        }

    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return {"method": "os.killpg(SIGKILL)", "pid": proc.pid, "ok": True}
    except ProcessLookupError:
        return {"method": "os.killpg(SIGKILL)", "pid": proc.pid, "ok": True}
    except OSError as exc:
        proc.kill()
        return {
            "method": "os.killpg(SIGKILL) then proc.kill fallback",
            "pid": proc.pid,
            "ok": False,
            "stderr": str(exc),
        }
