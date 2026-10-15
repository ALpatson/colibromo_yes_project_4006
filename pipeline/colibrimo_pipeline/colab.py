"""Train on Google Colab after COLMAP on the laptop (D-020).

The laptop runs validate -> extract_frames -> poses, then zips only what
training needs (``export_for_colab``). The notebook
(``pipeline/notebooks/colab_run.ipynb``) unzips it, continues the run with
gsplat (train + export) and zips the result (``pack_result``). Back on the
laptop, ``import_result`` checks that result against the room's own COLMAP
model before putting it in the run folder.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from colibrimo_pipeline.check import check_files, describe
from colibrimo_pipeline.errors import PipelineError
from colibrimo_pipeline.manifest import MANIFEST_NAME, Manifest
from colibrimo_pipeline.utils.ply import read_vertex_count

FOR_COLAB_SUFFIX = "_for_colab.zip"
RESULT_SUFFIX = "_colab_result.zip"
STEPS_BEFORE_TRAINING = ("validate", "extract_frames", "poses")

# Files of the result: name in the zip (any folder) -> place in the run folder.
RESULT_FILES = {
    "scene.ply": "scene.ply",
    "cameras.json": "cameras.json",
    "manifest.json": MANIFEST_NAME,
    "transform.json": "train/transform.json",
    "train.log": "logs/train.log",
    "export.log": "logs/export.log",
}
REQUIRED_RESULT_FILES = ("scene.ply", "cameras.json", "manifest.json")
MAX_RESULT_FILE_BYTES = 4 * 1024**3


def _write_zip(zip_path: Path, files: list[tuple[Path, str]]) -> dict[str, Any]:
    """Zip (file, name in zip) pairs; JPEGs are stored (already compressed).

    Written to a ``.part`` file first, so a half-written zip is never taken for
    a finished one.
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(partial, "w") as archive:
        for path, name in files:
            stored = path.suffix.lower() in (".jpg", ".jpeg", ".png", ".ply")
            archive.write(
                path,
                name,
                compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
            )
    partial.replace(zip_path)
    return {
        "path": str(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "num_files": len(files),
    }


def files_for_colab(run_dir: Path) -> list[tuple[Path, str]]:
    """What training needs from a run folder: the manifest and the outputs of the
    steps before training (frames, frames.json, COLMAP model and database).

    Leaves out logs, candidate frames and unused COLMAP models.
    """
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST_NAME).is_file():
        raise PipelineError(f"{run_dir} has no {MANIFEST_NAME}: run it first.")
    manifest = Manifest.load_or_create(run_dir)
    for step in STEPS_BEFORE_TRAINING:
        if not manifest.is_step_done(step):
            raise PipelineError(
                f"Step '{step}' is not finished in this run: run up to COLMAP "
                "(camera poses) first."
            )
    files = {run_dir / MANIFEST_NAME}
    for step in STEPS_BEFORE_TRAINING:
        for rel in manifest.data["steps"][step].get("outputs", {}).values():
            path = run_dir / rel
            if path.is_dir():
                files.update(p for p in path.rglob("*") if p.is_file())
            else:
                files.add(path)
    return [(p, p.relative_to(run_dir).as_posix()) for p in sorted(files)]


def export_for_colab(run_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Zip what Colab needs to train this run (see files_for_colab)."""
    return _write_zip(zip_path, files_for_colab(run_dir))


def pack_result(run_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Zip a finished run's result for the laptop (run in the Colab notebook)."""
    run_dir = Path(run_dir)
    files = [
        (run_dir / rel, rel)
        for rel in RESULT_FILES.values()
        if (run_dir / rel).is_file()
    ]
    present = {Path(rel).name for _, rel in files}
    missing = [name for name in REQUIRED_RESULT_FILES if name not in present]
    if missing:
        raise PipelineError(f"The run has no {', '.join(missing)}: did it finish?")
    return _write_zip(zip_path, files)


def extract_result_zip(zip_path: Path, dest: Path) -> list[str]:
    """Copy the result files out of a zip into ``dest``; returns their names.

    Only the known file names are taken (whatever folder they are in inside
    the zip); everything else is ignored, so nothing can be written outside
    ``dest``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    found: list[str] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename.replace("\\", "/")).name
                if member.is_dir() or name not in RESULT_FILES:
                    continue
                if name in found:
                    raise PipelineError(f"The zip contains {name} twice.")
                if member.file_size > MAX_RESULT_FILE_BYTES:
                    raise PipelineError(f"{name} in the zip is too large.")
                with archive.open(member) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                found.append(name)
    except zipfile.BadZipFile as exc:
        raise PipelineError(f"Not a valid zip file: {exc}") from exc
    return found


def import_result(run_dir: Path, staging: Path) -> dict[str, Any]:
    """Check a result trained elsewhere (files in ``staging``) and put it in the run.

    Refused, with nothing changed, unless it was trained from this run's own
    COLMAP result and passes the coordinate and camera checks.
    """
    run_dir, staging = Path(run_dir), Path(staging)
    missing = [n for n in REQUIRED_RESULT_FILES if not (staging / n).is_file()]
    if missing:
        raise PipelineError(f"Missing from the result: {', '.join(missing)}.")
    local_path = run_dir / MANIFEST_NAME
    if not local_path.is_file():
        raise PipelineError("This room has no run yet: run it up to COLMAP first.")
    local = json.loads(local_path.read_text(encoding="utf-8"))
    try:
        incoming = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        cameras = json.loads((staging / "cameras.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Could not read the result's JSON files: {exc}") from exc

    steps = incoming.get("steps", {})
    if steps.get("export", {}).get("status") != "succeeded":
        raise PipelineError("The result's run did not finish (export did not succeed).")
    local_poses = local.get("steps", {}).get("poses", {}).get("finished_at")
    if not local_poses or steps.get("poses", {}).get("finished_at") != local_poses:
        raise PipelineError(
            "This result was not trained from this room's current COLMAP result "
            "(different camera poses). Export for Colab again and retrain."
        )
    export_info = steps["export"].get("info", {})
    if read_vertex_count(staging / "scene.ply") != export_info.get("num_gaussians"):
        raise PipelineError("scene.ply does not match the result's manifest.")

    result = check_files(run_dir, incoming, staging / "scene.ply", cameras)
    if not result["ok"]:
        raise PipelineError(
            "The result failed the checks, so it was not imported:\n"
            + "\n".join(describe(result))
        )

    for name, rel in RESULT_FILES.items():
        if name != "manifest.json" and (staging / name).is_file():
            target = run_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / name, target)
    manifest = Manifest(run_dir, incoming)
    manifest.set_run_info(
        imported_result={
            "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "check": result,
        }
    )
    manifest.save()
    shutil.rmtree(staging, ignore_errors=True)
    return result
