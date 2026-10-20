"""Score a trained splat on held-out frames: PSNR, SSIM, LPIPS (D-021).

Every ``train.test_every``-th frame (sorted by name, as gsplat does) is kept out
of training. After training, the exported splat is rendered at those frames and
compared with the real ones (``steps/_evaluate_helper.py``, run with the GPU
Python that has gsplat). Results go to ``<run>/eval/eval.json`` and the train
step's ``info.evaluation`` in the manifest; side-by-side images (real |
rendered) of ``train.eval_images`` fixed frames go to ``<run>/eval/``.

Rendering needs gsplat with CUDA (Colab, Linux GPU machine). A splat trained
elsewhere (e.g. Brush on Windows) is scored later with ``cpipe evaluate``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from colibrimo_pipeline.errors import PipelineError
from colibrimo_pipeline.manifest import MANIFEST_NAME, Manifest
from colibrimo_pipeline.utils.colmap_model import read_cameras, read_images
from colibrimo_pipeline.utils.proc import run_capture

EVAL_HELPER = Path(__file__).parent / "steps" / "_evaluate_helper.py"
EVAL_DIR = "eval"


def test_frame_names(sparse_model: Path, test_every: int) -> list[str]:
    """Held-out frames: every ``test_every``-th placed frame, sorted by name
    (frames 0, N, 2N, ... exactly as gsplat v1.5.3 splits them). 0 = none."""
    if test_every <= 0:
        return []
    names = sorted(image["name"] for image in read_images(sparse_model / "images.bin"))
    return names[::test_every]


def frames_to_save(test_frames: list[str], count: int) -> list[str]:
    """``count`` fixed frames spread evenly over the held-out ones (so over every
    video: names sort by video)."""
    if count <= 0 or not test_frames:
        return []
    if count >= len(test_frames):
        return list(test_frames)
    step = (len(test_frames) - 1) / (count - 1) if count > 1 else 0
    return [test_frames[round(i * step)] for i in range(count)]


def pinhole_cameras(sparse_model: Path) -> dict[str, dict[str, Any]]:
    """Each COLMAP camera as a plain pinhole K (distortion ignored) and its frame
    size: what a web viewer renders with (see the viewer's snapshot mode)."""
    cameras = {}
    for camera_id, cam in read_cameras(Path(sparse_model) / "cameras.bin").items():
        p = cam["params"]
        fx, fy, cx, cy = (
            (p[0], p[0], p[1], p[2])
            if cam["model"].startswith(("SIMPLE_", "RADIAL"))
            else p[:4]
        )
        cameras[str(camera_id)] = {
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "width": int(cam["width"]),
            "height": int(cam["height"]),
        }
    return cameras


def run_evaluation(
    run_dir: Path,
    ply: Path,
    sparse_model: Path,
    matrix: list[list[float]],
    test_frames: list[str],
    *,
    python: str,
    examples_dir: Path,
    antialiased: bool = False,
    eval_images: int = 5,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Render ``ply`` at the held-out frames and score it; returns the summary
    (the per-frame scores are in ``<run>/eval/eval.json``)."""
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    job = {
        "examples_dir": str(examples_dir),
        "data_dir": str(Path(sparse_model).parent.parent),
        "ply": str(ply),
        "matrix_4x4": matrix,
        "test_frames": test_frames,
        "save_frames": frames_to_save(test_frames, eval_images),
        "out_dir": str(out_dir),
        "antialiased": antialiased,
        "pinhole": pinhole_cameras(sparse_model),
    }
    job_file = out_dir / "job.json"
    job_file.write_text(json.dumps(job, indent=1), encoding="utf-8")
    if logger:
        logger.info(
            "Scoring the splat on %d held-out frames (PSNR, SSIM, LPIPS)...",
            len(test_frames),
        )
    result = run_capture([python, str(EVAL_HELPER), str(job_file)], timeout=3600)
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        raise PipelineError(
            f"Evaluation failed:\n{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    scores = json.loads(lines[-1])
    (out_dir / "eval.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    summary = {
        k: v for k, v in scores.items() if k not in ("per_frame", "saved_images")
    }
    summary["images"] = [f"{EVAL_DIR}/{Path(p).name}" for p in scores["saved_images"]]
    if logger:
        logger.info("Score: %s.", describe(summary))
    return summary


def evaluate_existing_run(
    run_dir: Path, python: str | None = None, examples_dir: Path | None = None
) -> dict[str, Any]:
    """Score a finished run's ``scene.ply`` (e.g. trained with Brush elsewhere) and
    record it in its manifest. Needs the GPU environment (gsplat + CUDA)."""
    from colibrimo_pipeline.gsplat_compat import apply_gsplat_fixes

    run_dir = Path(run_dir).resolve()
    if not (run_dir / MANIFEST_NAME).is_file():
        raise PipelineError(f"{run_dir} has no {MANIFEST_NAME}.")
    manifest = Manifest.load_or_create(run_dir)
    train = manifest.data.get("steps", {}).get("train", {})
    test_frames = train.get("info", {}).get("test_frames")
    if not test_frames:
        raise PipelineError(
            "This run kept no frames out of training (train.test_every 0, or trained "
            "before D-021), so it cannot be scored fairly."
        )
    matrix = (manifest.data.get("transform") or {}).get("matrix_4x4")
    if matrix is None or not (run_dir / "scene.ply").is_file():
        raise PipelineError("The run is not exported yet (no scene.ply / transform).")
    examples = examples_dir or os.environ.get("GSPLAT_EXAMPLES_DIR")
    if not examples:
        raise PipelineError(
            "Set GSPLAT_EXAMPLES_DIR to gsplat v1.5.3's examples folder."
        )
    python = python or sys.executable
    apply_gsplat_fixes(Path(examples), python)
    train_cfg = manifest.data.get("config", {}).get("train", {})
    summary = run_evaluation(
        run_dir,
        run_dir / "scene.ply",
        manifest.step_output("poses", "sparse_model"),
        matrix,
        test_frames,
        python=python,
        examples_dir=Path(examples),
        antialiased=bool(train_cfg.get("antialiased")),
        eval_images=int(train_cfg.get("eval_images", 5)),
    )
    train.setdefault("info", {})["evaluation"] = summary
    manifest.save()
    return summary


def describe(summary: dict[str, Any]) -> str:
    return (
        f"PSNR {summary['psnr']:.2f} dB, SSIM {summary['ssim']:.4f}, LPIPS "
        f"{summary['lpips']:.4f} on {summary['num_test_frames']} held-out frames "
        f"({summary['num_splats']:,} splats, SH {summary['sh_degree']}, "
        f"{summary['image_size'][0]}x{summary['image_size'][1]})"
    )
