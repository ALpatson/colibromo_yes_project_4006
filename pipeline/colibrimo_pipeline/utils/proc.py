"""Find and run external programs (ffmpeg, ffprobe, colmap, the gsplat trainer).

Tool output is streamed line by line into the logs. The console only gets a
short "heartbeat" with the latest output line every ``heartbeat_s`` seconds,
so long runs show progress without flooding the screen.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

from colibrimo_pipeline.errors import StepError, ToolNotFoundError

# Environment variables that can point to a tool's executable (useful on Windows
# when the tool is not on PATH, e.g. CPIPE_COLMAP=D:\tools\colmap\bin\colmap.exe).
TOOL_ENV_VARS = {
    "ffmpeg": "CPIPE_FFMPEG",
    "ffprobe": "CPIPE_FFPROBE",
    "colmap": "CPIPE_COLMAP",
}

INSTALL_HINTS = {
    "ffmpeg": "Install FFmpeg (https://ffmpeg.org/download.html) and add it to PATH.",
    "ffprobe": "ffprobe ships with FFmpeg (https://ffmpeg.org/download.html).",
    "colmap": "Install COLMAP (https://colmap.github.io/install.html) and add it to PATH.",
}

_LINE_SPLIT = re.compile(rb"[\r\n]")


def find_tool(name: str) -> str:
    """Return the path of an external tool, or raise a helpful ToolNotFoundError."""
    env_var = TOOL_ENV_VARS.get(name)
    if env_var and os.environ.get(env_var):
        path = os.environ[env_var]
        if Path(path).is_file():
            return path
        raise ToolNotFoundError(
            f"{env_var} is set to '{path}', but that file does not exist."
        )

    found = shutil.which(name)
    if found:
        return found

    hint = INSTALL_HINTS.get(name, "")
    extra = f" Or set {env_var} to its full path." if env_var else ""
    raise ToolNotFoundError(f"'{name}' was not found. {hint}{extra}")


def format_command(cmd: list[str]) -> str:
    """Readable, copy-pasteable version of a command for the logs."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_capture(
    cmd: list[str], timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Run a short command and capture its output (no logging, no error raising)."""
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def run_tool(
    cmd: list[str],
    *,
    step: str,
    logger: logging.Logger,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    heartbeat_s: float = 30.0,
    tail_lines: int = 25,
) -> list[str]:
    """Run ``cmd``, log its output, and raise StepError if it fails.

    Every output line goes to the log files (DEBUG level). Progress bars that
    redraw with carriage returns are split into separate lines too. Returns the
    last ``tail_lines`` lines of output.
    """
    cmd = [str(c) for c in cmd]
    logger.info("Running: %s", format_command(cmd))
    tool = Path(cmd[0]).name

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            bufsize=0,
        )
    except OSError as exc:
        raise StepError(step, f"Could not start {tool}: {exc}") from exc

    tail: deque[str] = deque(maxlen=tail_lines)
    latest = ""
    last_beat = time.monotonic()
    buffer = b""

    def handle(raw: bytes) -> None:
        nonlocal latest
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.debug("%s", line)
            tail.append(line)
            latest = line

    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.read(4096), b""):
        buffer += chunk
        *complete, buffer = _LINE_SPLIT.split(buffer)
        for raw in complete:
            handle(raw)
        now = time.monotonic()
        if latest and now - last_beat >= heartbeat_s:
            logger.info("  ... %s", latest[:200])
            last_beat = now
    handle(buffer)

    code = proc.wait()
    if code != 0:
        last = "\n".join(tail) or "(no output)"
        raise StepError(step, f"{tool} exited with code {code}. Last output:\n{last}")
    return list(tail)
