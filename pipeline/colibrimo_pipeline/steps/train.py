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
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from colibrimo_pipeline.check import coordinate_check
from colibrimo_pipeline.config import TrainConfig
from colibrimo_pipeline.errors import PipelineError, StepError
from colibrimo_pipeline.evaluate import EVAL_DIR, run_evaluation, test_frame_names
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
    find_model_dirs,
    move_unused_models,
    read_images,
    read_points3d_xyz,
    write_images_subset,
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


def _plain_flag(name: str) -> str:
    return f"--{name}"


def build_train_command(
    python: str,
    examples_dir: Path,
    data_dir: Path,
    result_dir: Path,
    cfg: TrainConfig,
    flag=_plain_flag,
) -> list[str]:
    """Command line for gsplat's simple_trainer.py (flags checked against gsplat v1.5.3).

    ``flag`` spells the optional flags (see trainer_flag_speller)."""
    steps = str(cfg.max_steps)
    # Checkpoints (D-023): gsplat also saves the splat + stats (time, GPU memory) there.
    saved = [str(s) for s in saved_steps(cfg)]
    options: list[str] = []
    if cfg.strategy == "mcmc" and cfg.max_splats:
        options += [flag("strategy.cap_max"), str(cfg.max_splats)]
    if cfg.antialiased:
        options.append(flag("antialiased"))
    if cfg.bilateral_grid:
        options.append(flag("use_bilateral_grid"))
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
        *saved,
        "--ply_steps",
        *saved,
        "--save_ply",
        "--disable_viewer",
        "--disable_video",
        *options,
        *cfg.extra_args,
    ]


def trainer_flag_speller(python: str, examples_dir: Path, strategy: str):
    """How this simple_trainer.py spells its flags (tyro: ``--a_b`` or ``--a-b``),
    read from its ``--help``; a flag it lacks is an error now, not after training."""
    result = run_capture(
        [python, str(examples_dir / "simple_trainer.py"), strategy, "--help"],
        timeout=300,
    )
    known = set(re.findall(r"--[\w.-]+", result.stdout + result.stderr))

    def flag(name: str) -> str:
        for spelling in (f"--{name}", f"--{name.replace('_', '-')}"):
            if spelling in known:
                return spelling
        raise StepError(
            NAME, f"gsplat's simple_trainer.py ({strategy}) has no option --{name}."
        )

    return flag


def expected_ply(result_dir: Path, max_steps: int) -> Path:
    # simple_trainer.py saves at `step == max_steps - 1` (0-based step counter).
    return result_dir / "ply" / f"point_cloud_{max_steps - 1}.ply"


def saved_steps(cfg: TrainConfig) -> list[int]:
    """Steps where gsplat saves the splat: the checkpoints, then the last step."""
    return [*cfg.checkpoint_steps, cfg.max_steps]


def read_train_stats(result_dir: Path, step: int) -> dict:
    """gsplat's stats at a saved step: elapsed training time, peak GPU memory, splats.

    simple_trainer.py writes ``stats/train_step<step-1>_rank0.json`` with ``mem``
    (GB, torch.cuda.max_memory_allocated), ``ellipse_time`` (s since training
    started) and ``num_GS`` at every save step. Empty dict if missing.
    """
    path = result_dir / "stats" / f"train_step{step - 1:04d}_rank0.json"
    if not path.is_file():
        return {}
    stats = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train_seconds": round(float(stats["ellipse_time"]), 1),
        "gpu_peak_mem_gb": round(float(stats["mem"]), 2),
        "num_gaussians": int(stats["num_GS"]),
    }


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
    # A result imported from Colab is replaced by this training (D-020).
    ctx.manifest.data.pop("imported_result", None)
    if ctx.config.train.trainer == "brush":
        return run_brush(ctx)
    return run_gsplat(ctx)


def run_brush(ctx: StepContext) -> StepResult:
    """Train with Brush (any GPU, Windows too). Coordinates stay COLMAP's (identity)."""
    cfg = ctx.config.train
    if cfg.checkpoint_steps:
        ctx.logger.warning(
            "Warning: train.checkpoint_steps is gsplat only; Brush scores only its "
            "final splat (use cpipe evaluate on Colab)."
        )
    exe = brush_executable(ctx.config)
    sparse_model = ctx.manifest.step_output("poses", "sparse_model")  # colmap/sparse/0
    data_dir = sparse_model.parent.parent  # colmap/

    # Brush scans the whole colmap/ folder and mixes files of every model it finds
    # (testnew: images of one leftover model, points of another; D-018). Move
    # leftovers of older run folders out, then make sure only sparse/0 is left.
    moved = move_unused_models(sparse_model.parent)
    for name in moved:
        ctx.logger.info("Moved unused COLMAP model %s to %s/.", name, UNUSED_MODELS_DIR)
    models = find_model_dirs(data_dir)
    if [m.resolve() for m in models] != [sparse_model.resolve()]:
        found = [m.relative_to(data_dir).as_posix() for m in models] or ["none"]
        raise StepError(
            NAME,
            "Brush must see exactly one COLMAP model (colmap/sparse/0), but colmap/ "
            f"contains: {', '.join(found)}. Move the others out of colmap/ "
            f"(e.g. to {UNUSED_MODELS_DIR}/) and resume.",
        )

    result_dir = ctx.run_dir / "train"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True)
    test_frames = _record_test_frames(ctx, sparse_model, result_dir)
    if test_frames:  # Brush gets a copy of the dataset without the held-out frames
        data_dir = _dataset_without(
            sparse_model, set(test_frames), result_dir / "brush_data"
        )

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
        "Training %d steps with Brush (SH degree %d, max resolution %s, max splats %s), %s...",
        cfg.max_steps,
        cfg.sh_degree,
        cfg.max_resolution or ctx.config.frames.max_long_side,
        cfg.max_splats or "Brush default",
        _holdout_text(test_frames),
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
            "test_every": cfg.test_every,
            "test_frames": test_frames,
            # Scoring renders with gsplat (CUDA): run `cpipe evaluate` on Colab.
            "evaluation": None,
        },
        versions={"brush": brush_version(exe)},
    )


def _dataset_without(sparse_model: Path, held_out: set[str], dest: Path) -> Path:
    """A Brush dataset (images/ + sparse/0) without the held-out frames."""
    images = sparse_model.parent.parent / "images"
    (dest / "images").mkdir(parents=True)
    model = dest / "sparse" / "0"
    model.mkdir(parents=True)
    keep = {i["name"] for i in read_images(sparse_model / "images.bin")} - held_out
    for name in sorted(keep):
        try:
            os.link(images / name, dest / "images" / name)
        except OSError:
            shutil.copy2(images / name, dest / "images" / name)
    write_images_subset(sparse_model / "images.bin", model / "images.bin", keep)
    for other in sparse_model.iterdir():
        if other.is_file() and other.name != "images.bin":
            shutil.copy2(other, model / other.name)
    return dest


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

    sparse_model = ctx.manifest.step_output("poses", "sparse_model")  # colmap/sparse/0
    data_dir = sparse_model.parent.parent  # colmap/
    result_dir = ctx.run_dir / "train"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True)
    test_frames = _record_test_frames(ctx, sparse_model, result_dir)

    needs_options = (
        cfg.antialiased
        or cfg.bilateral_grid
        or (cfg.strategy == "mcmc" and cfg.max_splats)
    )
    flag = (
        trainer_flag_speller(python, examples_dir, cfg.strategy)
        if needs_options
        else _plain_flag
    )
    command = build_train_command(
        python, examples_dir, data_dir, result_dir, cfg, flag=flag
    )
    ctx.logger.info(
        "Training %d steps with gsplat (%s strategy%s%s%s), %s...",
        cfg.max_steps,
        cfg.strategy,
        (
            f", max {cfg.max_splats:,} splats"
            if cfg.strategy == "mcmc" and cfg.max_splats
            else ""
        ),
        ", anti-aliased" if cfg.antialiased else "",
        ", per-frame colour correction" if cfg.bilateral_grid else "",
        _holdout_text(test_frames),
    )
    start = time.monotonic()
    run_tool(
        command,
        step=NAME,
        logger=ctx.logger,
        cwd=examples_dir,
        heartbeat_s=60,
    )
    train_seconds = time.monotonic() - start

    ply = expected_ply(result_dir, cfg.max_steps)
    if not ply.is_file():
        raise StepError(NAME, f"Training finished but {ply.name} was not written.")

    transform = compute_transform(python, examples_dir, data_dir, cfg.test_every)
    # Same check as for Brush, after gsplat's normalisation: the transform we
    # record must be the one gsplat trained in (cameras.json is built with it).
    check = coordinate_check(
        read_points3d_xyz(sparse_model / "points3D.bin"),
        read_vertex_xyz(ply),
        np.asarray(transform["matrix_4x4"]),
    )
    ctx.logger.info(
        "Coordinate check: median distance from a transformed COLMAP point to the "
        "nearest splat = %.4f x scene size (pass if < %.3f).",
        check["median_nearest_splat_rel"],
        check["threshold_rel"],
    )
    if not check["same_coordinates"]:
        raise StepError(
            NAME,
            "The gsplat splat is not where the recorded scene transform puts COLMAP's "
            "points, so cameras.json would not match scene.ply. See docs/decisions.md "
            f"D-020. Check: {check}",
            info={"coordinate_check": check},
        )
    transform["coordinate_check"] = check
    transform_path = result_dir / "transform.json"
    transform_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")

    num_gaussians = read_vertex_count(ply)
    ctx.logger.info("Trained %d Gaussians in %.0f s.", num_gaussians, train_seconds)

    evaluation: dict = {}
    if test_frames:
        try:
            evaluation = run_evaluation(
                ctx.run_dir,
                ply,
                sparse_model,
                transform["matrix_4x4"],
                test_frames,
                python=python,
                examples_dir=examples_dir,
                antialiased=cfg.antialiased,
                eval_images=cfg.eval_images,
                logger=ctx.logger,
            )
        except (
            PipelineError
        ) as exc:  # keep the trained splat; say why there is no score
            ctx.logger.warning("Warning: %s", exc)
            evaluation = {"error": str(exc)[-500:]}
    checkpoints = _score_checkpoints(
        ctx,
        result_dir,
        sparse_model,
        transform["matrix_4x4"],
        test_frames,
        python=python,
        examples_dir=examples_dir,
    )
    final_stats = read_train_stats(result_dir, cfg.max_steps)
    return StepResult(
        outputs={"ply": ply, "transform": transform_path},
        info={
            "trainer": "gsplat simple_trainer.py",
            "strategy": cfg.strategy,
            "max_steps": cfg.max_steps,
            "sh_degree": cfg.sh_degree,
            "max_splats": cfg.max_splats if cfg.strategy == "mcmc" else None,
            "antialiased": cfg.antialiased,
            "bilateral_grid": cfg.bilateral_grid,
            "seed": TRAINER_SEED,
            "num_gaussians": num_gaussians,
            "ply_size_bytes": ply.stat().st_size,
            "train_seconds": round(train_seconds, 1),
            "gpu": {"name": env_versions.get("gpu")},
            "coordinate_check": check,
            "test_every": cfg.test_every,
            "test_frames": test_frames,
            "evaluation": evaluation or None,
            "gpu_peak_mem_gb": final_stats.get("gpu_peak_mem_gb"),
            "checkpoints": checkpoints or None,
        },
        versions={"gpu_env": env_versions},
    )


def _score_checkpoints(
    ctx: StepContext,
    result_dir: Path,
    sparse_model: Path,
    matrix: list[list[float]],
    test_frames: list[str],
    *,
    python: str,
    examples_dir: Path,
) -> list[dict]:
    """Score the splats gsplat saved at ``train.checkpoint_steps`` (D-023).

    One entry per checkpoint: step, training time and peak GPU memory at that
    step, splats, file size and the same scores as the final splat (or why not).
    """
    cfg = ctx.config.train
    checkpoints = []
    for step in cfg.checkpoint_steps:
        entry: dict = {"step": step, **read_train_stats(result_dir, step)}
        ply = expected_ply(result_dir, step)
        if not ply.is_file():
            entry["error"] = f"{ply.name} was not written"
            checkpoints.append(entry)
            continue
        entry["num_gaussians"] = read_vertex_count(ply)
        entry["ply_size_bytes"] = ply.stat().st_size
        if test_frames:
            ctx.logger.info("Checkpoint at step %d:", step)
            try:
                entry["evaluation"] = run_evaluation(
                    ctx.run_dir,
                    ply,
                    sparse_model,
                    matrix,
                    test_frames,
                    python=python,
                    examples_dir=examples_dir,
                    antialiased=cfg.antialiased,
                    eval_images=cfg.eval_images,
                    logger=ctx.logger,
                    out_subdir=f"{EVAL_DIR}/step_{step}",
                )
            except PipelineError as exc:
                ctx.logger.warning("Warning: %s", exc)
                entry["evaluation"] = {"error": str(exc)[-500:]}
        checkpoints.append(entry)
    return checkpoints


def _record_test_frames(
    ctx: StepContext, sparse_model: Path, result_dir: Path
) -> list[str]:
    """The held-out frames (train.test_every), saved in train/test_frames.json."""
    test_frames = test_frame_names(sparse_model, ctx.config.train.test_every)
    (result_dir / "test_frames.json").write_text(
        json.dumps({"test_every": ctx.config.train.test_every, "frames": test_frames}),
        encoding="utf-8",
    )
    return test_frames


def _holdout_text(test_frames: list[str]) -> str:
    if not test_frames:
        return "on all frames (no held-out frames, no score)"
    return f"keeping {len(test_frames)} frames out of training to score the result"
