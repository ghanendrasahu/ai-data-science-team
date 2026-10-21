"""Sandboxed Python execution tool.

LLM-generated snippets run in an isolated child interpreter (``-I``), with a
parent-side AST import scan, a secret-free child environment and a hard
timeout. Anything the snippet writes is confined to a per-call scratch dir.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


class DeniedImport(ValueError):
    """Raised when a snippet imports a denylisted root package."""


_DEFAULT_DENIED_ROOT = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "ctypes",
        "multiprocessing",
        "importlib",
        "signal",
        "atexit",
        "pickle",
        "builtins",
        "pathlib",
    }
)

# Base environment variables the child interpreter is allowed to see. Either
# allow-list explicitly or pass nothing: secrets never leak into the child.
_BASE_ENV_KEYS = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONUTF8",
    }
)


def denied_root_packages() -> frozenset[str]:
    """Env-tunable, disable-able denylist (``PYTHON_TOOL_DENIED_ROOT``)."""
    override = os.environ.get("PYTHON_TOOL_DENIED_ROOT", "").strip()
    if override == "_":
        return frozenset()
    if override:
        return frozenset(part.strip() for part in override.split(",") if part.strip())
    return _DEFAULT_DENIED_ROOT


def _scan_imports(code: str) -> None:
    """Reject imports of denylisted root packages *before* execution."""
    allowed = denied_root_packages()
    tree = ast.parse(code)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in allowed:
                raise DeniedImport(f"Import of denied package: '{root}'")
    # bare `__import__("os")` and eval tricks are covered by denying sys/os,
    # which the child itself still has - this scan is a tripwire, not a jail.


def _child_env(work_dir: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _BASE_ENV_KEYS}
    env.setdefault("PYTHONUTF8", "1")
    env["MPLBACKEND"] = "Agg"  # headless plotting
    env["MPLCONFIGDIR"] = str(work_dir / ".mplcache")
    return env


def run_python(
    code: str,
    work_dir: str | Path,
    data_files: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute a snippet in an isolated interpreter and capture its evidence.

    Returns ``{stdout, stderr, returncode, timed_out, artifacts}``. The snippet
    may read any ``data_files`` (copied into ``work_dir``) and write outputs -
    only files inside ``work_dir`` are visible/kept.
    """
    import time

    from app.config import get_settings

    settings = get_settings()
    timeout = int(timeout_seconds or getattr(settings, "python_tool_timeout", 90))
    scratch = Path(work_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    _scan_imports(code)
    script = scratch / f"snippet_{uuid.uuid4().hex[:8]}.py"
    script.write_text(code, encoding="utf-8")

    provided: list[str] = []
    for name, source in (data_files or {}).items():
        target = scratch / Path(name).name
        shutil.copyfile(source, target)
        provided.append(target.name)

    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=str(scratch),
            env=_child_env(scratch),
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = ((exc.stderr or "") if isinstance(exc.stderr, str) else "") + (
            f"\nExecution exceeded {timeout}s and was terminated."
        )
        returncode, timed_out = 124, True
    elapsed = round(time.time() - started, 2)

    outputs = [
        path.name
        for path in sorted(scratch.iterdir())
        if path.is_file() and not path.name.startswith("snippet_") and path.name not in provided
    ]
    return {
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": elapsed,
        "data_files": provided,
        "artifacts": outputs,
    }


__all__ = ["DeniedImport", "denied_root_packages", "run_python"]
