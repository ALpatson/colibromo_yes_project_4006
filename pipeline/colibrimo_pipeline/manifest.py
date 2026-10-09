"""Run manifest: ``manifest.json`` in every run folder.

Records the input video, resolved configuration, tool versions, git commit,
and for each step its status, timings, outputs and results. It is also how the
pipeline knows which steps are already done (resumability).

Output paths are stored relative to the run folder, so a run folder can be
moved between machines (e.g. poses computed on a laptop, training on Colab).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Manifest:
    """In-memory view of ``manifest.json``. Call :meth:`save` to write it."""

    def __init__(self, run_dir: Path, data: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.data = data

    @property
    def path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @classmethod
    def load_or_create(cls, run_dir: Path) -> Manifest:
        path = Path(run_dir) / MANIFEST_NAME
        if path.is_file():
            return cls(run_dir, json.loads(path.read_text(encoding="utf-8")))
        return cls(
            run_dir,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "new",
                "created_at": now_iso(),
                "steps": {},
                "tool_versions": {},
                "outputs": {},
            },
        )

    # ---- paths -------------------------------------------------------------

    def rel(self, path: Path) -> str:
        """Path relative to the run folder, with forward slashes."""
        return Path(path).resolve().relative_to(self.run_dir.resolve()).as_posix()

    def abs(self, rel_path: str) -> Path:
        return self.run_dir / rel_path

    # ---- run level ---------------------------------------------------------

    def set_run_info(self, **fields: Any) -> None:
        """Top-level fields such as input, config, pipeline_version, git_commit."""
        self.data.update(fields)

    def add_tool_versions(self, versions: dict[str, Any]) -> None:
        self.data.setdefault("tool_versions", {}).update(versions)

    def set_status(self, status: str, error: str | None = None) -> None:
        self.data["status"] = status
        if error is None:
            self.data.pop("error", None)
        else:
            self.data["error"] = error

    # ---- step level --------------------------------------------------------

    def step(self, name: str) -> dict[str, Any]:
        return self.data.setdefault("steps", {}).setdefault(name, {"status": "pending"})

    def step_info(self, name: str) -> dict[str, Any]:
        """Results recorded by a finished step (empty dict if none)."""
        return self.data.get("steps", {}).get(name, {}).get("info", {})

    def step_output(self, name: str, key: str) -> Path:
        """Absolute path of an output recorded by a finished step."""
        return self.abs(self.data["steps"][name]["outputs"][key])

    def mark_started(self, name: str) -> None:
        self.data.setdefault("steps", {})[name] = {
            "status": "running",
            "started_at": now_iso(),
        }

    def mark_succeeded(
        self,
        name: str,
        outputs: dict[str, Path],
        info: dict[str, Any],
        duration_s: float,
    ) -> None:
        step = self.step(name)
        step.update(
            status="succeeded",
            finished_at=now_iso(),
            duration_s=round(duration_s, 2),
            outputs={key: self.rel(path) for key, path in outputs.items()},
            info=info,
        )
        step.pop("error", None)

    def mark_failed(self, name: str, error: str, duration_s: float) -> None:
        self.step(name).update(
            status="failed",
            finished_at=now_iso(),
            duration_s=round(duration_s, 2),
            error=error,
        )

    def is_step_done(self, name: str) -> bool:
        """True if the step succeeded before and all its outputs still exist."""
        step = self.data.get("steps", {}).get(name)
        if not step or step.get("status") != "succeeded":
            return False
        return all(self.abs(p).exists() for p in step.get("outputs", {}).values())

    # ---- persistence -------------------------------------------------------

    def save(self) -> None:
        """Write atomically (temp file + rename) so a crash never leaves half a file."""
        self.data["updated_at"] = now_iso()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
