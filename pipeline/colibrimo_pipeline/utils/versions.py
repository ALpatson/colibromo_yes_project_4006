"""Collect versions of everything that influences a run (for the manifest)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from colibrimo_pipeline.utils.proc import run_capture

PACKAGE_NAME = "colibrimo-pipeline"

# Run with the trainer's Python: reports torch / CUDA / gsplat / GPU as JSON.
_GPU_ENV_SCRIPT = """
import json, platform
info = {"python": platform.python_version()}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
except Exception as exc:
    info["torch_error"] = str(exc)
try:
    import gsplat
    info["gsplat"] = gsplat.__version__
except Exception as exc:
    info["gsplat_error"] = str(exc)
print(json.dumps(info))
"""


def pipeline_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        from colibrimo_pipeline import __version__

        return __version__


def git_commit() -> str | None:
    """Commit of the pipeline code.

    From CPIPE_GIT_COMMIT (set in the Docker image), pip's install record
    (``pip install git+...``), or the git checkout the code runs from.
    """
    if os.environ.get("CPIPE_GIT_COMMIT"):
        return os.environ["CPIPE_GIT_COMMIT"]
    try:
        direct_url = metadata.distribution(PACKAGE_NAME).read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except OSError:
        return None


def system_info() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}


def first_output_line(cmd: list[str], must_contain: str = "") -> str | None:
    """First non-empty output line of ``cmd`` (containing ``must_contain``), or None."""
    try:
        result = run_capture(cmd, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        line = line.strip()
        if line and must_contain.lower() in line.lower():
            return line
    return None


def ffmpeg_version(ffmpeg: str) -> str | None:
    return first_output_line([ffmpeg, "-version"], "version")


def colmap_version(colmap: str) -> str | None:
    # COLMAP 4.x has `colmap version`; older versions print it at the top of `colmap help`.
    return first_output_line([colmap, "version"], "colmap") or first_output_line(
        [colmap, "help"], "colmap"
    )


def gpu_env_versions(python: str | None = None) -> dict[str, object]:
    """torch / CUDA / gsplat / GPU versions as seen by ``python`` (default: this one)."""
    try:
        result = run_capture(
            [python or sys.executable, "-c", _GPU_ENV_SCRIPT], timeout=180
        )
        lines = result.stdout.strip().splitlines()
        return (
            json.loads(lines[-1]) if lines else {"error": result.stderr.strip()[-500:]}
        )
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"error": str(exc)}
