"""Step 4: train the Gaussian Splatting scene with gsplat (needs an NVIDIA GPU).

Runs gsplat's ``examples/simple_trainer.py`` (gsplat v1.5.3) instead of
Nerfstudio's ``ns-train splatfacto`` (see docs/decisions.md, D-005). The
trainer is headless (no viewer), uses a fixed random seed (42, set inside
simple_trainer.py) and writes the PLY at the last step.

Output layout::

    train/ply/point_cloud_<max_steps-1>.ply   trained splat
    train/ckpts/ckpt_<max_steps-1>_rank0.pt   checkpoint (for evaluation, Step 2)
    train/transform.json                      normalisation applied by gsplat
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from colibrimo_pipeline.config import TrainConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps.base import StepContext, StepResult
from colibrimo_pipeline.utils.ply import read_vertex_count
from colibrimo_pipeline.utils.proc import run_capture, run_tool
from colibrimo_pipeline.utils.versions import gpu_env_versions

NAME = "train"
TRAINER_SEED = (
    42  # hard-coded in gsplat's simple_trainer.py (set_random_seed(42 + rank))
)
TRANSFORM_HELPER = Path(__file__).with_name("_gsplat_transform.py")

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


def make_datasets_importable(examples_dir: Path) -> bool:
    """Make gsplat's ``examples/datasets`` folder win over an installed ``datasets`` package.

    gsplat v1.5.3 ships ``examples/datasets/`` without ``__init__.py``, so Python
    treats it as a namespace package. If the unrelated HuggingFace ``datasets``
    package is installed (it is on Google Colab), Python imports that instead and
    ``from datasets.colmap import ...`` fails. An empty ``__init__.py`` turns the
    folder into a regular package, found first because the script's folder is at
    the front of ``sys.path``. Returns True if the file was created.
    """
    init_file = examples_dir / "datasets" / "__init__.py"
    if init_file.parent.is_dir() and not init_file.exists():
        init_file.write_text(
            "# Added by colibrimo_pipeline: see steps/train.py make_datasets_importable\n",
            encoding="utf-8",
        )
        return True
    return False


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
    cfg = ctx.config.train
    python = cfg.python or sys.executable
    examples_dir = gsplat_examples_dir(cfg)
    if make_datasets_importable(examples_dir):
        ctx.logger.info("Added datasets/__init__.py to gsplat's examples (import fix).")

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
