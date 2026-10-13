"""Step 4: train the Gaussian Splatting scene with gsplat or Brush (``train.trainer``).

gsplat (default): runs gsplat's ``examples/simple_trainer.py`` (gsplat v1.5.3)
instead of Nerfstudio's ``ns-train splatfacto`` (see docs/decisions.md, D-005);
needs Linux + an NVIDIA GPU. Headless, fixed seed (42, set inside
simple_trainer.py), writes the PLY at the last step::

    train/ply/point_cloud_<max_steps-1>.ply   trained splat
    train/ckpts/ckpt_<max_steps-1>_rank0.pt   checkpoint (for evaluation, Step 2)
    train/transform.json                      normalisation applied by gsplat

Brush (D-013): runs ``brush_app`` headless; works on Windows with any GPU::

    train/export_<max_steps>.ply              trained splat
    train/transform.json                      identity (Brush keeps COLMAP's frame)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from colibrimo_pipeline.config import TrainConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.gsplat_compat import apply_gsplat_fixes
from colibrimo_pipeline.steps._brush import (
    brush_executable,
    brush_version,
    build_brush_command,
    expected_brush_ply,
    identity_check,
    parse_adapter,
    parse_progress,
)
from colibrimo_pipeline.steps.base import StepContext, StepResult
from colibrimo_pipeline.utils.colmap_model import (
    UNUSED_MODELS_DIR,
    move_unused_models,
    read_points3d_xyz,
)
from colibrimo_pipeline.utils.ply import read_vertex_count, read_vertex_xyz
from colibrimo_pipeline.utils.proc import run_capture, run_tool
from colibrimo_pipeline.utils.versions import gpu_env_versions

NAME = "train"
TRAINER_SEED = (
    42  # hard-coded in gsplat's simple_trainer.py (set_random_seed(42 + rank))
)
TRANSFORM_HELPER = Path(__file__).with_name("_gsplat_transform.py")
BRUSH_CACHE_DIR = Path.home() / ".cache" / "colibrimo-pipeline" / "brush"

NO_GPU_MESSAGE = (
    "Training needs an NVIDIA GPU with CUDA, which this environment does not have "
    "(or PyTorch/gsplat are not installed). Run the first steps here with "
    "'--until poses', then continue the same run folder on a GPU machine or Google "
    "Colab (pipeline/notebooks/colab_run.ipynb). See docs/pipeline.md."
)


def gsplat_examples_dir(cfg: TrainConfig) -> Path:
    """Folder containing gsplat's simple_trainer.py (config, then GSPLAT_EXAMPLES_DIR)."""
    configured = cfg.gsplat_examples_dir or os.environ.get("GSPLAT_EXAMPLES_DIR")
    if not configured:
        raise StepError(
            NAME,
            "gsplat's examples folder is not configured. Set GSPLAT_EXAMPLES_DIR (or "
            "train.gsplat_examples_dir) to the 'examples' folder of a gsplat v1.5.3 checkout. "
            "See docs/pipeline.md.",
        )
    path = Path(configured)
    if not (path / "simple_trainer.py").is_file():
        raise StepError(
            NAME, f"No simple_trainer.py in '{path}'. Is this gsplat's examples folder?"
        )
    return path


def build_train_command(
    python: str, examples_dir: Path, data_dir: Path, result_dir: Path, cfg: TrainConfig
) -> list[str]:
    """Command line for gsplat's simple_trainer.py (flags checked against gsplat v1.5.3)."""
    steps = str(cfg.max_steps)
    return [
        python,
        str(examples_dir / "simple_trainer.py"),
        cfg.strategy,
        "--data_dir",
        str(data_dir),
        "--data_factor",
        "1",  # frames were already downscaled in extract_frames
        "--result_dir",
        str(result_dir),
        "--max_steps",
        steps,
        "--test_every",
        str(cfg.test_every),
        "--sh_degree",
        str(cfg.sh_degree),
        "--eval_steps",
        "-1",  # evaluation is a separate step (PRD Step 2)
        "--save_steps",
        steps,
        "--ply_steps",
        steps,
        "--save_ply",
        "--disable_viewer",
        "--disable_video",
        *cfg.extra_args,
    ]


def expected_ply(result_dir: Path, max_steps: int) -> Path:
    # simple_trainer.py saves at `step == max_steps - 1` (0-based step counter).
    return result_dir / "ply" / f"point_cloud_{max_steps - 1}.ply"


def compute_transform(
    python: str, examples_dir: Path, data_dir: Path, test_every: int
) -> dict:
    result = run_capture(
        [
            python,
            str(TRANSFORM_HELPER),
            str(examples_dir),
            str(data_dir),
            str(test_every),
        ],
        timeout=600,
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        raise StepError(
            NAME,
            f"Could not compute the scene normalisation:\n{result.stderr.strip()[-1500:]}",
        )
    return json.loads(lines[-1])


def run(ctx: StepContext) -> StepResult:
    if ctx.config.train.trainer == "brush":
        return run_brush(ctx)
    return run_gsplat(ctx)


def run_brush(ctx: StepContext) -> StepResult:
    """Train with Brush (any GPU, Windows too). Coordinates stay COLMAP's (identity)."""
    cfg = ctx.config.train
    exe = brush_executable(ctx.config)
    sparse_model = ctx.manifest.step_output("poses", "sparse_model")  # colmap/sparse/0
    data_dir = sparse_model.parent.parent  # colmap/

    # Brush loads whatever model it finds in sparse/: make sure only "0" is there
    # (run folders made before this was done by the poses step).
    moved = move_unused_models(sparse_model.parent)
    for name in moved:
        ctx.logger.info(
            "Moved unused COLMAP model sparse/%s to %s/.", name, UNUSED_MODELS_DIR
        )

    result_dir = ctx.run_dir / "train"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True)

    adapter: dict[str, str] = {}

    def watch(line: str) -> None:
        if not adapter and (found := parse_adapter(line)):
            adapter.update(found)
            ctx.logger.info(
                "Brush GPU: %s (%s, %s)",
                found["name"],
                found["device_type"],
                found["backend"],
            )
            if found["device_type"] != "DiscreteGpu":
                ctx.logger.warning(
                    "Warning: Brush is not using a dedicated GPU (%s). On a laptop with "
                    "two GPUs, set brush_app.exe to 'High performance' in Windows Settings "
                    "> System > Display > Graphics.",
                    found["name"],
                )
        # Progress on the console every 2000 steps (Brush logs every 200).
        if (progress := parse_progress(line)) and progress[0] // 2000 > last_report[0]:
            last_report[0] = progress[0] // 2000
            ctx.logger.info(
                "  Brush: step %d / %d, %d splats",
                progress[0],
                cfg.max_steps,
                progress[1],
            )

    last_report = [0]
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "info")  # Brush then logs its GPU adapter and progress
    ctx.logger.info(
        "Training %d steps with Brush (SH degree %d, max resolution %s, max splats %s)...",
        cfg.max_steps,
        cfg.sh_degree,
        cfg.max_resolution or ctx.config.frames.max_long_side,
        cfg.max_splats or "Brush default",
    )
    start = time.monotonic()
    # Brush keeps a GPU kernel-tuning cache (target/autotune) in its working
    # folder: use a persistent one so later runs reuse it.
    BRUSH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    run_tool(
        build_brush_command(exe, data_dir, result_dir, ctx.config),
        step=NAME,
        logger=ctx.logger,
        cwd=BRUSH_CACHE_DIR,
        env=env,
        heartbeat_s=60,
        on_line=watch,
    )
    train_seconds = time.monotonic() - start

    ply = expected_brush_ply(result_dir, cfg.max_steps)
    if not ply.is_file():
        found = sorted(p.name for p in result_dir.glob("*.ply"))
        raise StepError(
            NAME,
            f"Brush finished but {ply.name} was not written (found: {found or 'none'}).",
        )

    # Brush does not normalise the scene: check it, then record the identity.
    check = identity_check(
        read_points3d_xyz(sparse_model / "points3D.bin"), read_vertex_xyz(ply)
    )
    ctx.logger.info(
        "Coordinate check: median distance from a COLMAP point to the nearest splat "
        "= %.4f x scene size (identity if < %.3f).",
        check["median_nearest_splat_rel"],
        check["threshold_rel"],
    )
    if not check["same_coordinates"]:
        raise StepError(
            NAME,
            "The Brush splat does not seem to be in COLMAP's coordinates (this Brush "
            "version may normalise the scene). The scene transform cannot be recorded "
            f"as identity; see docs/decisions.md D-013. Check: {check}",
            info={"identity_check": check},
        )
    transform = {
        "matrix_4x4": np.eye(4).tolist(),
        "scene_scale": 1.0,
        "normalisation": "none: Brush keeps COLMAP's coordinates",
        "identity_check": check,
    }
    transform_path = result_dir / "transform.json"
    transform_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")

    num_gaussians = read_vertex_count(ply)
    ctx.logger.info("Trained %d Gaussians in %.0f s.", num_gaussians, train_seconds)
    return StepResult(
        outputs={"ply": ply, "transform": transform_path},
        info={
            "trainer": "brush",
            "max_steps": cfg.max_steps,
            "sh_degree": cfg.sh_degree,
            "max_resolution": cfg.max_resolution or ctx.config.frames.max_long_side,
            "max_splats": cfg.max_splats,
            "seed": 42,
            "gpu": adapter or None,
            "num_gaussians": num_gaussians,
            "ply_size_bytes": ply.stat().st_size,
            "train_seconds": round(train_seconds, 1),
            "moved_unused_models": moved,
        },
        versions={"brush": brush_version(exe)},
    )


def run_gsplat(ctx: StepContext) -> StepResult:
    cfg = ctx.config.train
    python = cfg.python or sys.executable
    examples_dir = gsplat_examples_dir(cfg)
    for change in apply_gsplat_fixes(examples_dir, python):
        ctx.logger.info("gsplat compatibility fix: %s", change)

    env_versions = gpu_env_versions(python)
    if not env_versions.get("cuda_available"):
        raise StepError(NAME, NO_GPU_MESSAGE, info={"environment": env_versions})
    ctx.logger.info(
        "GPU: %s (%s GB), torch %s, CUDA %s, gsplat %s",
        env_versions.get("gpu"),
        env_versions.get("gpu_memory_gb"),
        env_versions.get("torch"),
        env_versions.get("cuda"),
        env_versions.get("gsplat"),
    )

    data_dir = ctx.manifest.step_output(
        "poses", "sparse_model"
    ).parent.parent  # colmap/
    result_dir = ctx.run_dir / "train"
    shutil.rmtree(result_dir, ignore_errors=True)

    ctx.logger.info(
        "Training %d steps with gsplat (%s strategy)...", cfg.max_steps, cfg.strategy
    )
    run_tool(
        build_train_command(python, examples_dir, data_dir, result_dir, cfg),
        step=NAME,
        logger=ctx.logger,
        cwd=examples_dir,
        heartbeat_s=60,
    )

    ply = expected_ply(result_dir, cfg.max_steps)
    if not ply.is_file():
        raise StepError(NAME, f"Training finished but {ply.name} was not written.")

    transform = compute_transform(python, examples_dir, data_dir, cfg.test_every)
    transform_path = result_dir / "transform.json"
    transform_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")

    num_gaussians = read_vertex_count(ply)
    ctx.logger.info("Trained %d Gaussians.", num_gaussians)
    return StepResult(
        outputs={"ply": ply, "transform": transform_path},
        info={
            "trainer": "gsplat simple_trainer.py",
            "strategy": cfg.strategy,
            "max_steps": cfg.max_steps,
            "sh_degree": cfg.sh_degree,
            "seed": TRAINER_SEED,
            "num_gaussians": num_gaussians,
            "ply_size_bytes": ply.stat().st_size,
        },
        versions={"gpu_env": env_versions},
    )
